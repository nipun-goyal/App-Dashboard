
/****** Script for SelectTopNRows command from SSMS  ******/
SELECT p.DataID, p.pdfText, p.pdfText_Nouns_Lemma, a.Path, a.[Submitter Type], a.[Submitter City], a.[Submitter Organization],
    a.[Submitter Title], a.[Submitter Company], a.[Document Type], a.[Filing Date], a.Commodity 
        FROM dbo.pdfContent p
            LEFT JOIN dbo.app_dash a ON p.DataID = a.DataID;